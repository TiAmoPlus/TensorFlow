package com.example.cifar10;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Looper;
import android.os.MessageQueue;
import android.os.PersistableBundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.app.AppCompatDelegate;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.target.Target;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadFactory;

public class MainActivity extends AppCompatActivity {
    // 模型相关配置
    private static final int PICTURE_REQUEST_CODE=10000;
    private static final int TAKE_PHOTO_REQUEST_CODE=10001;
    private static final int CAMERA_PERMISSIONS_REQUEST_CODE = 119;
    private static final int PERMISSIONS_REQUEST = 108;
    private static final int OPEN_SETTING_REQUEST_COED = 110;
    private static final String CURRENT_TAKE_PHOTO_URI = "currentTakePhotoUri";
    private static final String TAG="TAG";
    private static final String PACKAGE_URL_SCHEME = "package:";

    private static final int INPUT_SIZE = 32;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "x_input";
    private static final String OUTPUT_NAME = "final_result";
    private static final String MODEL_FILE = "file:///android_asset/model/model.pb";
    private static final String LABEL_FILE = "file:///android_asset/model/imagenet_comp_graph_label_strings_china.txt";

    private Executor executor;
    private Uri currentTakePhotoUri;

    private TextView result;
    private ImageView ivPicture;
    private ImageView choosePicture;
    private ImageView takePhoto;
    private Classifier classifier;
    static {
        AppCompatDelegate.setCompatVectorFromResourcesEnabled(true);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //isTaskRoot()方法是用来判断当前Activity是否是第一个activity
        if(!isTaskRoot()){
            finish();
        }
        setContentView(R.layout.activity_main);
        //申请存储权限
        String[] permissions={
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA,
                Manifest.permission.INTERNET,
                Manifest.permission.ACCESS_NETWORK_STATE
        };
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,permissions,321);
        }
        if (ContextCompat.checkSelfPermission(this,Manifest.permission.INTERNET)!=PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,permissions,321);
        }
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,permissions,321);
        }
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_NETWORK_STATE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,permissions,321);
        }
        choosePicture=findViewById(R.id.iv_choose_picture);
        choosePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, PICTURE_REQUEST_CODE);
            }
        });
        takePhoto=findViewById(R.id.iv_take_photo);
        takePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSIONS_REQUEST_CODE);
                } else {
                    openSystemCamera();
                }
            }
        });
        ivPicture = findViewById(R.id.iv_picture);
        result = findViewById(R.id.tv_classifier_info);
        // 避免耗时任务占用 CPU 时间片造成UI绘制卡顿，提升启动页面加载速度
        Looper.myQueue().addIdleHandler(idleHandler);
    }
    /**
     *  主线程消息队列空闲时（视图第一帧绘制完成时）处理耗时事件
     */
    MessageQueue.IdleHandler idleHandler = new MessageQueue.IdleHandler() {
        @Override
        public boolean queueIdle() {
            // 初始化 Classifier
            if (classifier == null) {
                // 创建 TensorFlowImageClassifier
                classifier = TensorFlowImageClassifier.create(MainActivity.this.getAssets(),
                        MODEL_FILE, LABEL_FILE, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME);
            }

            // 初始化线程池
            executor = new ScheduledThreadPoolExecutor(1, new ThreadFactory() {
                @Override
                public Thread newThread(@NonNull Runnable r) {
                    Thread thread = new Thread(r);
                    thread.setDaemon(true);
                    thread.setName("ThreadPool-ImageClassifier");
                    return thread;
                }
            });
            // 请求权限
            requestMultiplePermissions();
            // 返回 false 时只会回调一次
            return false;
        }
    };
    /**
     * 打开系统相机
     */
    private void openSystemCamera() {
        //调用系统相机
        Intent takePhotoIntent = new Intent();
        takePhotoIntent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);

        //这句作用是如果没有相机则该应用不会闪退，要是不加这句则当系统没有相机应用的时候该应用会闪退
        if (takePhotoIntent.resolveActivity(getPackageManager()) == null) {
            Toast.makeText(this, "当前系统没有可用的相机应用", Toast.LENGTH_SHORT).show();
            return;
        }

        String fileName = "TF_" + System.currentTimeMillis() + ".jpg";
        File photoFile = new File(FileUtil.getPhotoCacheFolder(), fileName);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            //通过FileProvider创建一个content类型的Uri
            currentTakePhotoUri = FileProvider.getUriForFile(this, "gdut.bsx.tensorflowtraining.fileprovider", photoFile);
            //对目标应用临时授权该 Uri 所代表的文件
            takePhotoIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } else {
            currentTakePhotoUri = Uri.fromFile(photoFile);
        }

        //将拍照结果保存至 outputFile 的Uri中，不保留在相册中
        takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, currentTakePhotoUri);
        startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
    }
    private void handleInputPhoto(Uri imageUri) {
        // 加载图片

        Glide.with(MainActivity.this).asBitmap().listener(new RequestListener<Bitmap>() {

            @Override
            public boolean onLoadFailed(@Nullable GlideException e, Object model, Target<Bitmap> target, boolean isFirstResource) {
                Log.d(TAG,"handleInputPhoto onLoadFailed");
                Toast.makeText(MainActivity.this, "图片加载失败", Toast.LENGTH_SHORT).show();
                return false;
            }

            @Override
            public boolean onResourceReady(Bitmap resource, Object model, Target<Bitmap> target, DataSource dataSource, boolean isFirstResource) {
                Log.d(TAG,"handleInputPhoto onResourceReady");
                startImageClassifier(resource);
                return false;
            }
        }).load(imageUri).into(ivPicture);

        result.setText("Processing...");

    }
    /**
     * 开始图片识别匹配
     * @param bitmap
     */
    private void startImageClassifier(final Bitmap bitmap) {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    Log.i(TAG, Thread.currentThread().getName() + " startImageClassifier");
                    Bitmap croppedBitmap = getScaleBitmap(bitmap, INPUT_SIZE);
                    final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
                    Log.i(TAG, "startImageClassifier results: " + results);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            result.setText(String.format("results: %s", results));
                        }
                    });
                } catch (IOException e) {
                    Log.e(TAG, "startImageClassifier getScaleBitmap " + e.getMessage());
                }
            }
        });
    }
    /**
     * 对图片进行缩放
     * @param bitmap
     * @param size
     * @return
     * @throws IOException
     */
    private static Bitmap getScaleBitmap(Bitmap bitmap, int size) throws IOException {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float) size) / width;
        float scaleHeight = ((float) size) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }
    /**
     * 请求相机和外部存储权限
     */
    private void requestMultiplePermissions() {

        String storagePermission = Manifest.permission.WRITE_EXTERNAL_STORAGE;
        String cameraPermission = Manifest.permission.CAMERA;

        int hasStoragePermission = ActivityCompat.checkSelfPermission(this, storagePermission);
        int hasCameraPermission = ActivityCompat.checkSelfPermission(this, cameraPermission);

        List<String> permissions = new ArrayList<>();
        if (hasStoragePermission != PackageManager.PERMISSION_GRANTED) {
            permissions.add(storagePermission);
        }

        if (hasCameraPermission != PackageManager.PERMISSION_GRANTED) {
            permissions.add(cameraPermission);
        }

        if (!permissions.isEmpty()) {
            String[] params = permissions.toArray(new String[permissions.size()]);
            ActivityCompat.requestPermissions(this, params, PERMISSIONS_REQUEST);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == PICTURE_REQUEST_CODE) {
                // 处理选择的图片
                handleInputPhoto(data.getData());
            } else if (requestCode == OPEN_SETTING_REQUEST_COED){
                requestMultiplePermissions();
            } else if (requestCode == TAKE_PHOTO_REQUEST_CODE) {
                // 如果拍照成功，加载图片并识别
                handleInputPhoto(currentTakePhotoUri);
            }
        }
    }

    @Override
    public void onSaveInstanceState(Bundle outState, PersistableBundle outPersistentState) {

        outState.putParcelable(CURRENT_TAKE_PHOTO_URI,currentTakePhotoUri);
        super.onSaveInstanceState(outState, outPersistentState);
    }

    @Override
    protected void onRestoreInstanceState(Bundle savedInstanceState) {
        super.onRestoreInstanceState(savedInstanceState);
        if (savedInstanceState != null) {
            currentTakePhotoUri = savedInstanceState.getParcelable(CURRENT_TAKE_PHOTO_URI);
        }
    }
    /**
     *  显示缺失权限提示，可再次请求动态权限
     */
    private void showNeedStoragePermissionDialog() {
        new AlertDialog.Builder(this)
                .setTitle("权限获取提示")
                .setMessage("必须要有存储权限才能获取到图片")
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                    }
                })
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        ActivityCompat.requestPermissions(MainActivity.this,
                                new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST);
                    }
                }).setCancelable(false)
                .show();
    }


    /**
     *  显示权限被拒提示，只能进入设置手动改
     */
    private void showMissingStoragePermissionDialog() {
        new AlertDialog.Builder(this)
                .setTitle("权限获取失败")
                .setMessage("必须要有存储权限才能正常运行")
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        MainActivity.this.finish();
                    }
                })
                .setPositiveButton("去设置", new DialogInterface.OnClickListener() {
                    @Override public void onClick(DialogInterface dialog, int which) {
                        startAppSettings();
                    }
                })
                .setCancelable(false)
                .show();
    }

    private void showNeedCameraPermissionDialog() {
        new AlertDialog.Builder(this)
                .setMessage("摄像头权限被关闭，请开启权限后重试")
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                })
                .create().show();
    }



    /**
     * 启动应用的设置进行授权
     */
    private void startAppSettings() {
        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.setData(Uri.parse(PACKAGE_URL_SCHEME + getPackageName()));
        startActivityForResult(intent, OPEN_SETTING_REQUEST_COED);
    }

}
